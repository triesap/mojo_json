// simdjson native benchmark - symmetric methodology with the json mojo
// bench:
//
//   - 3 warmup iterations + 100 measured iterations
//   - Two paths: parse_only (`parser.parse(s)`) and parse_traverse
//     (parse + walk every node, accessing each leaf value)
//   - Reports min/avg/max wall time and min-time-derived throughput
//
// Throughput is computed from the *minimum* iteration time so it is
// directly comparable to the json mojo bench, which reports the same
// statistic (the simdjson community convention).
//
// Compile: clang++ -O3 -std=c++17 -o bench_simdjson bench_simdjson.cpp \
//              -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -lsimdjson

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include "simdjson.h"

// Walk the DOM, touching every leaf's value (string, number, bool).
// This mirrors what the mojo bench's `_walk` does, so parse_traverse
// numbers on both sides measure the same kind of work.
static size_t traverse_element(simdjson::dom::element elem)
{
    size_t count = 1;
    switch (elem.type())
    {
    case simdjson::dom::element_type::ARRAY:
        for (auto child : elem.get_array().value())
        {
            count += traverse_element(child);
        }
        break;
    case simdjson::dom::element_type::OBJECT:
        for (auto [key, value] : elem.get_object().value())
        {
            (void)key;
            count += traverse_element(value);
        }
        break;
    case simdjson::dom::element_type::STRING:
        (void)elem.get_string().value();
        break;
    case simdjson::dom::element_type::INT64:
        (void)elem.get_int64().value();
        break;
    case simdjson::dom::element_type::UINT64:
        (void)elem.get_uint64().value();
        break;
    case simdjson::dom::element_type::DOUBLE:
        (void)elem.get_double().value();
        break;
    case simdjson::dom::element_type::BOOL:
        (void)elem.get_bool().value();
        break;
    default:
        break;
    }
    return count;
}

struct Stats
{
    double min_ms;
    double avg_ms;
    double max_ms;
};

static Stats summarize(const std::vector<double> &times)
{
    Stats s{times[0], 0.0, times[0]};
    for (double t : times)
    {
        s.min_ms = std::min(s.min_ms, t);
        s.max_ms = std::max(s.max_ms, t);
        s.avg_ms += t;
    }
    s.avg_ms /= static_cast<double>(times.size());
    return s;
}

static void print_row(const char *label,
                      const Stats &s,
                      size_t file_size)
{
    double throughput = (file_size / 1e9) / (s.min_ms / 1000.0);
    std::cout << "  " << label
              << ": min " << s.min_ms << " ms"
              << " | avg " << s.avg_ms << " ms"
              << " | max " << s.max_ms << " ms"
              << " | " << throughput << " GB/s (min-based)"
              << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <json_file>" << std::endl;
        return 1;
    }

    std::string filepath = argv[1];

    std::ifstream file(filepath);
    if (!file)
    {
        std::cerr << "Error: Cannot open file " << filepath << std::endl;
        return 1;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_str = buffer.str();
    size_t file_size = json_str.size();

    std::cout << "\n--- simdjson native (C++) ---" << std::endl;
    std::cout << "File: " << filepath << std::endl;
    std::cout << "Size: " << file_size << " bytes ("
              << (file_size / 1024.0) << " KB)" << std::endl;
    std::cout << std::endl;

    // simdjson reuses one parser instance across iterations, which
    // amortizes its internal scratch buffer allocations. The mojo
    // bench builds a fresh `Document` per iteration, so this is the
    // closest like-for-like setup we can offer the C++ side.
    simdjson::dom::parser parser;

    constexpr int num_warmup = 3;
    constexpr int num_iters = 100;

    // Warmup (both paths).
    for (int i = 0; i < num_warmup; i++)
    {
        auto doc = parser.parse(json_str);
        (void)traverse_element(doc.value());
    }

    // ---- parse_only ----
    std::vector<double> parse_only_times;
    parse_only_times.reserve(num_iters);
    for (int i = 0; i < num_iters; i++)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto doc = parser.parse(json_str);
        // Touch the root tag so the optimizer cannot elide the parse.
        (void)doc.value().type();
        auto t1 = std::chrono::high_resolution_clock::now();
        parse_only_times.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    // ---- parse_traverse ----
    std::vector<double> parse_traverse_times;
    parse_traverse_times.reserve(num_iters);
    size_t total_nodes = 0;
    for (int i = 0; i < num_iters; i++)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto doc = parser.parse(json_str);
        total_nodes = traverse_element(doc.value());
        auto t1 = std::chrono::high_resolution_clock::now();
        parse_traverse_times.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    auto parse_only = summarize(parse_only_times);
    auto parse_traverse = summarize(parse_traverse_times);

    std::cout << "Iterations: " << num_iters
              << " (warmup " << num_warmup << ")" << std::endl;
    std::cout << "Nodes:      " << total_nodes << std::endl;
    std::cout << std::endl;

    print_row("parse_only    ", parse_only, file_size);
    print_row("parse_traverse", parse_traverse, file_size);
    std::cout << std::endl;

    return 0;
}

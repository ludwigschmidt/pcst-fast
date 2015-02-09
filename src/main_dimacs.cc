#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <locale>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <unistd.h>

#include "pcst_fast.h"

using namespace std;

using cluster_approx::PCSTFast;

struct Options {
  int num_trials;
  bool lenient_parsing;
  string tree_output_filename;
  string json_output_filename;
  string stp_output_filename;
};

struct RunningTimeStatistics {
  double total_time;
  double reading_time;
  vector<double> algo_setup_times;
  vector<double> algo_main_times;
  vector<double> algo_total_times;
};

enum ProblemType {
  kUnknownProblemType = 0,
  kPCSPG,
  kRPCST,
  kMWCS
};

bool parse_options(Options* options, int argc, char** argv);


bool read_input_basic(vector<pair<int, int>>* edges,
                      vector<double>* prizes,
                      vector<double>* costs);

bool read_input_STP(vector<pair<int, int>>* edges,
                    vector<double>* prizes,
                    vector<double>* costs,
                    int* root,
                    string* instance_name,
                    ProblemType* problem_type,
                    bool lenient_parsing);

string get_problem_type_string(ProblemType problem_type);

double get_min_prize(const vector<double>& prizes);

void convert_mwcs_to_pcst(const vector<double>& prizes,
                          const vector<double>& costs,
                          vector<double>* pcst_prizes,
                          vector<double>* pcst_costs);

void output_function(const char* msg);

void solution_cost(const vector<double>& prizes,
                   const vector<double>& costs,
                   const vector<int>& result_nodes,
                   const vector<int>& result_edges,
                   double* node_cost,
                   double* edge_cost,
                   double* dual_bound);

void solution_cost_mwcs(const vector<double>& prizes,
                        const vector<int>& result_nodes,
                        double* node_cost,
                        double* dual_bound);

void build_json_array(const vector<double>& data, string* output);

double get_mean_without_outliers(const vector<double>& data, int num_outliers);

bool check_connectivity(int n,
                        vector<pair<int, int>> graph_edges,
                        vector<int> result_edges,
                        vector<int> result_nodes);

bool write_tree_output(const string& filename,
                       const vector<pair<int, int>>& edges,
                       const vector<int>& tree_edges);

bool write_json_output(const string& filename,
                       const Options& opts,
                       const vector<pair<int, int>>& edges,
                       const vector<double>& prizes,
                       const vector<double>& costs,
                       const vector<int>& result_nodes,
                       const vector<int>& result_edges,
                       const PCSTFast::Statistics& pcst_stats,
                       const RunningTimeStatistics rt_stats,
                       const string& instance_name,
                       ProblemType problem_type);

bool write_stp_output(const string& filename,
                      const Options& opts,
                      const vector<pair<int, int>>& edges,
                      const vector<double>& prizes,
                      const vector<double>& costs,
                      const vector<int>& result_nodes,
                      const vector<int>& result_edges,
                      const RunningTimeStatistics& rt_stats,
                      const string& instance_name,
                      ProblemType problem_type);


int main(int argc, char** argv) {
  auto total_start = chrono::high_resolution_clock::now();

  vector<pair<int, int>> edges;
  vector<double> prizes;
  vector<double> costs;
  int root = PCSTFast::kNoRoot;
  ProblemType problem_type = kUnknownProblemType;
  string instance_name;

  Options opts;
  if (!parse_options(&opts, argc, argv)) {
    fprintf(stderr, "Could not read options.\n");
    return 1;
  }

  if (!read_input_STP(&edges,
                      &prizes,
                      &costs,
                      &root,
                      &instance_name,
                      &problem_type,
                      opts.lenient_parsing)) {
    fprintf(stderr, "Could not read input.\n");
    return 1;
  }

  RunningTimeStatistics rt_stats;
  vector<int> result_nodes;
  vector<int> result_edges;
  PCSTFast::Statistics pcst_stats;

  auto reading_stop = chrono::high_resolution_clock::now();
  chrono::duration<double> reading_time = reading_stop - total_start;
  rt_stats.reading_time = reading_time.count();

  vector<double> mwcs_modified_prizes;
  vector<double> mwcs_modified_costs;
  if (problem_type == kMWCS) {
    double m = get_min_prize(prizes);
    if (m >= 0.0) {
      fprintf(stderr, "Trivial MWCS instance: only non-negative weights.\n");
      return 1;
    }
    convert_mwcs_to_pcst(prizes,
                         costs,
                         &mwcs_modified_prizes,
                         &mwcs_modified_costs);
  }

  for (int ii = 0; ii < opts.num_trials; ++ii) {
    auto start = chrono::high_resolution_clock::now();

    PCSTFast algo(edges,
                  (problem_type == kMWCS ? mwcs_modified_prizes : prizes),
                  (problem_type == kMWCS ? mwcs_modified_costs : costs),
                  root,
                  (root == PCSTFast::kNoRoot ? 1 : 0),
                  PCSTFast::kStrongPruning,
                  0,
                  output_function);


    auto algo_main_start = chrono::high_resolution_clock::now();

    if (!algo.run(&result_nodes, &result_edges)) {
      fprintf(stderr, "Algorithm returned false.\n");
      return 1;
    }

    auto stop = chrono::high_resolution_clock::now();
    chrono::duration<double> algo_setup_time = algo_main_start - start;
    chrono::duration<double> algo_main_time = stop - algo_main_start;
    rt_stats.algo_setup_times.push_back(algo_setup_time.count());
    rt_stats.algo_main_times.push_back(algo_main_time.count());
    rt_stats.algo_total_times.push_back(algo_setup_time.count()
                                        + algo_main_time.count());
    
    algo.get_statistics(&pcst_stats);
  }

  auto total_stop = chrono::high_resolution_clock::now();
  chrono::duration<double> total_time = total_stop - total_start;
  rt_stats.total_time = total_time.count();

  if (!check_connectivity(prizes.size(), edges, result_edges, result_nodes)) {
    fprintf(stderr, "Algorithm did not return a tree.\n");
    return 1;
  }

  if (problem_type == kRPCST
      && find(result_nodes.begin(), result_nodes.end(), root)
          == result_nodes.end()) {
    fprintf(stderr, "Solution does not contain the root node.\n");
    return 1;
  }

  if (opts.tree_output_filename != "") {
    if (!write_tree_output(opts.tree_output_filename, edges, result_edges)) {
      fprintf(stderr, "Error while writing tree output file.\n");
      return 1;
    }
  }

  if (opts.json_output_filename != "") {
    if (!write_json_output(opts.json_output_filename,
                           opts,
                           edges,
                           prizes,
                           costs,
                           result_nodes,
                           result_edges,
                           pcst_stats,
                           rt_stats,
                           instance_name,
                           problem_type)) {
      fprintf(stderr, "Error while writing tree output file.\n");
      return 1;
    }
  }

  if (opts.stp_output_filename != "") {
    if (!write_stp_output(opts.stp_output_filename,
                          opts,
                          edges,
                          prizes,
                          costs,
                          result_nodes,
                          result_edges,
                          rt_stats,
                          instance_name,
                          problem_type)) {
      fprintf(stderr, "Error while writing tree output file.\n");
      return 1;
    }
  }

  return 0;
}


void solution_cost(const vector<double>& prizes,
                   const vector<double>& costs,
                   const vector<int>& result_nodes,
                   const vector<int>& result_edges,
                   double* node_cost,
                   double* edge_cost,
                   double* dual_bound) {
  *node_cost = 0;
  *edge_cost = 0;
  for (double v : prizes) {
    *node_cost += v;
  }
  for (int ii : result_nodes) {
    *node_cost -= prizes[ii];
  }
  for (int ii : result_edges) {
    *edge_cost += costs[ii];
  }
  *dual_bound = 0.5 * *edge_cost + *node_cost;
}


void solution_cost_mwcs(const vector<double>& prizes,
                        const vector<int>& result_nodes,
                        double* node_cost,
                        double* dual_bound) {
  *node_cost = 0.0;
  for (int ii : result_nodes) {
    *node_cost += prizes[ii];
  }
  double m = get_min_prize(prizes);
  *dual_bound = *node_cost - ((result_nodes.size() - 1.0) / 2.0) * m;
}


bool check_connectivity(int n,
                        vector<pair<int, int>> graph_edges,
                        vector<int> result_edges,
                        vector<int> result_nodes) {
  vector<vector<int>> neighbors(n);
  for (int edge_index : result_edges) {
    const pair<int, int>& edge = graph_edges[edge_index];
    neighbors[edge.first].push_back(edge.second);
    neighbors[edge.second].push_back(edge.first);
  }

  vector<bool> visited(n, false);
  queue<int> q;
  visited[result_nodes[0]] = true;
  q.push(result_nodes[0]);
  while (!q.empty()) {
    int cur = q.front();
    q.pop();
    for (int next : neighbors[cur]) {
      if (!visited[next]) {
        visited[next] = true;
        q.push(next);
      }
    }
  }
  for (int ii : result_nodes) {
    if (!visited[ii]) {
      return false;
    }
  }
  return true;
}


// File format
// First line:
// <number of nodes> <number of edges> <number of nonzero weight nodes
//                                                                  (terminals)
// One line per edge:
// <node1> <node2> <weight>
// node1 and node2 are indexed starting at 0. weight is in scientific notation.
// One line per terminal
// <node> <weight>
// node is indexed start at 0. weight is in scientific notation.
bool read_input_basic(vector<pair<int, int>>* edges,
                      vector<double>* prizes,
                      vector<double>* costs) {
  edges->clear();
  prizes->clear();
  costs->clear();
  int n, m, t;
  if (scanf("%d %d %d", &n, &m, &t) < 3) {
    return false;
  }
  for (int ii = 0; ii < m; ++ii) {
    int u, v;
    double val;
    if (scanf("%d %d %le", &u, &v, &val) < 3) {
      return false;
    }
    edges->push_back(make_pair(u, v));
    costs->push_back(val);
  }
  sort(edges->begin(), edges->end());
  prizes->resize(n, 0.0);
  for (int ii = 0; ii < t; ++ii) {
    int u;
    double val;
    if (scanf("%d %le", &u, &val) < 2) {
      return false;
    }
    (*prizes)[u] = val;
  }

  return true;
}

void ltrim(std::string* s) {
  s->erase(s->begin(), find_if(s->begin(),
                               s->end(),
                               not1(ptr_fun<int, int>(isspace))));
}

void rtrim(string* s) {
  s->erase(find_if(s->rbegin(),
                   s->rend(),
                   not1(ptr_fun<int, int>(isspace))).base(),
           s->end());
}


bool startswith(const string& s, const string& prefix) {
  return s.substr(0, prefix.length()) == prefix;
}


void remove_prefix(string* s, const string& prefix) {
  s->erase(s->begin(), s->begin() + prefix.length());
  ltrim(s);
}


bool read_input_STP(vector<pair<int, int>>* edges,
                    vector<double>* prizes,
                    vector<double>* costs,
                    int* root,
                    string* instance_name,
                    ProblemType* problem_type,
                    bool lenient_parsing) {
  const string first_line = "33D32945 STP File, STP Format Version 1.0";
  const string section_prefix = "SECTION";
  const string nodes_prefix = "Nodes";
  const string edges_prefix = "Edges";
  const string edge_prefix = "E";
  const string instance_name_prefix = "Name";
  const string problem_type_prefix = "Problem";
  const string terminals_prefix = "Terminals";
  const string terminal_prefix = "TP";
  const string terminal_prefix2 = "T";
  const string root_prefix = "RootP";
  enum SectionType {
    kComment,
    kGraph,
    kTerminals,
    kNoSection
  };
  map<string, SectionType> allowed_sections;
  allowed_sections["comment"] = kComment;
  allowed_sections["comments"] = kComment;
  allowed_sections["terminals"] = kTerminals;
  allowed_sections["graph"] = kGraph;
  map<string, ProblemType> problem_map;
  problem_map["rooted prize-collecting steiner problem in graphs"] = kRPCST;
  problem_map["prize-collecting steiner tree"] = kPCSPG;
  problem_map["prize-collecting steiner problem in graphs"] = kPCSPG;
  problem_map["maximum node weight connected subgraph"] = kMWCS;
  int lineno = 0;
  SectionType section = kNoSection;
  
  int n = -1, m = -1, t = -1;
  edges->clear();
  prizes->clear();
  costs->clear();
  *instance_name = "";
  *root = PCSTFast::kNoRoot;
  *problem_type = kUnknownProblemType;
  vector<pair<int, double>> terminals;
  vector<tuple<int, int, double>> tmp_edges;

  bool read_EOF = false;
  int num_edges_read = 0;
  int num_terminals_read = 0;
  string line = "";
  cin.sync_with_stdio(false);

  while (cin) {
    getline(cin, line);
    if (cin.eof()) {
      cerr << "Unexpected early end of file." << endl;
      cin.sync_with_stdio(true);
      return false;
    }

    lineno += 1;

    size_t hashpos = line.find('#');
    if (hashpos != string::npos) {
      line.erase(line.begin() + hashpos, line.end());
    }
    rtrim(&line);

    if (line == "") {
      continue;
    }

    if (lineno == 1) {
      if (line != first_line) {
        cerr << "Error: unexpected first line." << endl;
        cin.sync_with_stdio(true);
        return false;
      }
      continue;
    }

    if (line == "EOF") {
      if (section != kNoSection) {
        cerr << "Error: unexpected end of file, still in a section." << endl;
        cin.sync_with_stdio(true);
        return false;
      }
      read_EOF = true;
      break;
    }

    if (section == kNoSection) {
      if (!startswith(line, section_prefix)) {
        cerr << "Error: expected section start in line " << lineno << "."
             << endl;
        cin.sync_with_stdio(true);
        return false;
      }

      remove_prefix(&line, section_prefix);
      transform(line.begin(), line.end(), line.begin(), ::tolower);
      if (allowed_sections.find(line) == allowed_sections.end()) {
        cerr << "Error: unexpected section type \"" << line << "\" in line "
             << lineno << "." << endl;
        cin.sync_with_stdio(true);
        return false;
      }
      section = allowed_sections[line];
      continue;
    }

    if (line == "END") {
      section = kNoSection;
      continue;
    }

    if (section == kComment) {
      if (startswith(line, instance_name_prefix)) {
        remove_prefix(&line, instance_name_prefix);
        if (lenient_parsing && line[0] == ':') {
          line = line.substr(1, line.length() - 1);
          ltrim(&line);
        }
        if (line.length() < 2
            || line[0] != '\"'
            || line[line.length() - 1] != '\"') {
          cerr << "Error: instance name should be in quotation marks "
               << "(line " << lineno << ")." << endl;
          cin.sync_with_stdio(true);
          return false;
        }

        line = line.substr(1, line.length() - 2);
        if (line == "") {
          cerr << "Error: empty instance name in line " << lineno << "."
               << endl;
          cin.sync_with_stdio(true);
          return false;
        }

        if (*instance_name != "") {
          cerr << "Error: instance name appeared a second time in line "
               << lineno << "." << endl;
          cin.sync_with_stdio(true);
          return false;
        }

        *instance_name = line;
      } else if (startswith(line, problem_type_prefix)) {
        remove_prefix(&line, problem_type_prefix);
        if (line.length() < 2
            || line[0] != '\"'
            || line[line.length() - 1] != '\"') {
          cerr << "Error: problem type should be in quotation marks "
               << "(line " << lineno << ")." << endl;
          cin.sync_with_stdio(true);
          return false;
        }

        line = line.substr(1, line.length() - 2);
        transform(line.begin(), line.end(), line.begin(), ::tolower);

        if (problem_map.find(line) == problem_map.end()) {
          cerr << "Error: unexpected problem type \"" << line << "\" in line "
               << lineno << "." << endl;
          cin.sync_with_stdio(true);
          return false;
        }

        if (*problem_type != kUnknownProblemType) {
          cerr << "Error: problem type set a second time in line \"" << line
               << "\"." << endl;
          cin.sync_with_stdio(true);
          return false;
        }

        *problem_type = problem_map[line];
      }
      continue;
    }

    if (section == kGraph) {
      if (startswith(line, nodes_prefix)) {
        remove_prefix(&line, nodes_prefix);
        n = stoi(line);
      } else if (startswith(line, edges_prefix)) {
        remove_prefix(&line, edges_prefix);
        m = stoi(line);
      } else if (startswith(line, edge_prefix)) {
        remove_prefix(&line, edge_prefix);
        int node1, node2;
        double weight;
        istringstream tmp(line);
        tmp >> node1 >> node2 >> weight;
        tmp_edges.push_back(make_tuple(node1 - 1, node2 - 1, weight));
        num_edges_read += 1;
      } else {
        cerr << "Error: unexpected graph keyword in line " << lineno << "."
             << endl;
        cin.sync_with_stdio(true);
        return false;
      }
      continue;
    }

    if (section == kTerminals) {
      if (startswith(line, terminals_prefix)) {
        remove_prefix(&line, terminals_prefix);
        t = stoi(line);
      } else if (startswith(line, terminal_prefix)
          || startswith(line, terminal_prefix2)) {
        if (startswith(line, terminal_prefix)) {
          remove_prefix(&line, terminal_prefix);
        } else {
          remove_prefix(&line, terminal_prefix2);
        }
        int node;
        double weight;
        istringstream tmp(line);
        tmp >> node >> weight;
        terminals.push_back(make_pair(node - 1, weight));
        num_terminals_read += 1;
      } else if (startswith(line, root_prefix)) {
        remove_prefix(&line, root_prefix);
        if (*root != PCSTFast::kNoRoot) {
          cerr << "Error: root set a second time in line " << lineno << "."
               << endl;
          cin.sync_with_stdio(true);
          return false;
        }
        *root = stoi(line);
        *root -= 1;
        num_terminals_read += 1;
      } else {
        cerr << "Error: unexpected terminal keyword in line " << lineno << "."
             << endl;
        cin.sync_with_stdio(true);
        return false;
      }
      continue;
    }

    cerr << "Error: reached unexpected state in line " << lineno << "." << endl;
    cin.sync_with_stdio(true);
    return false;
  }
 
  if (!read_EOF) {
    cerr << "Error: did not read \"EOF\"." << endl;
    cin.sync_with_stdio(true);
    return false;
  }

  if (n == -1 || m == -1 || t == -1) {
    cerr << "Error: did not read number of nodes, edges, or terminals." << endl;
    cin.sync_with_stdio(true);
    return false;
  }

  if (m != num_edges_read) {
    cerr << "Error: read " << num_edges_read << " edges in total, not " << m
         << "." << endl;
    cin.sync_with_stdio(true);
    return false;
  }

  if (t != num_terminals_read) {
    cerr << "Error: read " << num_terminals_read << " terminals in total, not "
         << t << "." << endl;
    cin.sync_with_stdio(true);
    return false;
  }

  if (*instance_name == "") {
    cerr << "Error: did not read an instance name." << endl;
    cin.sync_with_stdio(true);
    return false;
  }

  if (*problem_type == kUnknownProblemType) {
    if (lenient_parsing) {
      *problem_type = kPCSPG;
    } else {
      cerr << "Error: problem type not set." << endl;
      cin.sync_with_stdio(true);
      return false;
    }
  }

  if (*problem_type == kRPCST) {
    if (*root == PCSTFast::kNoRoot) {
      cerr << "Error: root node not set." << endl;
      cin.sync_with_stdio(true);
      return false;
    }
  } else {
    if (*root != PCSTFast::kNoRoot) {
      cerr << "Error: root node set in an unrooted problem type." << endl;
      cin.sync_with_stdio(true);
      return false;
    }
  }

  for (size_t ii = 0; ii < tmp_edges.size(); ++ii) {
    int node1 = get<0>(tmp_edges[ii]);
    int node2 = get<1>(tmp_edges[ii]);
    if (node1 < 0 || node1 >= n || node2 < 0 || node2 >= n) {
      cerr << "Error: edge " << node1 + 1 << ", " << node2 + 1 << " is out of "
           << "the node range" << endl;
      cin.sync_with_stdio(true);
      return false;
    }

    if (node1 == node2) {
      cerr << "Error: edge " << node1 + 1 << ", " << node2 + 1 << " is a loop."
           << endl;
    }

    if (node1 > node2) {
      get<0>(tmp_edges[ii]) = node2;
      get<1>(tmp_edges[ii]) = node1;
    }
  }
  
  sort(tmp_edges.begin(), tmp_edges.end());
  for (size_t ii = 0; ii < tmp_edges.size() - 1; ++ii) {
    if (get<0>(tmp_edges[ii]) == get<0>(tmp_edges[ii + 1])
        && get<1>(tmp_edges[ii]) == get<1>(tmp_edges[ii + 1])) {
      cerr << "Error: edge " << get<0>(tmp_edges[ii]) + 1 << ", "
           << get<1>(tmp_edges[ii]) + 1 << " appears twice (edge pair was "
           << "normalized so that first_index < second_index)." << endl;
      cin.sync_with_stdio(true);
      return false;
    }
  }

  edges->resize(m);
  costs->resize(m);
  for (size_t ii = 0; ii < tmp_edges.size(); ++ii) {
    (*edges)[ii].first = get<0>(tmp_edges[ii]);
    (*edges)[ii].second = get<1>(tmp_edges[ii]);
    (*costs)[ii] = get<2>(tmp_edges[ii]);
  }

  prizes->resize(n, 0.0);
  for (size_t ii = 0; ii < terminals.size(); ++ii) {
    if (terminals[ii].first < 0 || terminals[ii].first >= n) {
      cerr << "Error: terminal " << terminals[ii].first + 1 << " is out of the "
           << "node range." << endl;
      cin.sync_with_stdio(true);
      return false;
    }
    (*prizes)[terminals[ii].first] = terminals[ii].second;
  }

  cin.sync_with_stdio(true);
  return true;
}


bool parse_options(Options* options, int argc, char** argv) {
  options->num_trials = 1;
  options->tree_output_filename = "";
  options->lenient_parsing = false;

  int c;
  while ((c = getopt(argc, argv, "j:lo:s:t:")) != -1) {
    if (c == 'o') {
      options->tree_output_filename = string(optarg);
    } else if (c == 'j') {
      options->json_output_filename = string(optarg);
    } else if (c == 'l') {
      options->lenient_parsing = true;
    } else if (c == 's') {
      options->stp_output_filename = string(optarg);
    } else if (c == 't') {
      options->num_trials = stoi(string(optarg));
    } else {
      return false;
    }
  }

  return true;
}


void build_json_array(const vector<double>& data, string* output) {
  ostringstream tmp;
  tmp << "[";
  for (size_t ii = 0; ii < data.size(); ++ii) {
    tmp << scientific << data[ii];
    if (ii != data.size() - 1) {
      tmp << ", ";
    }
  }
  tmp << "]";
  *output = tmp.str();
}


double get_mean_without_outliers(const vector<double>& data, int num_outliers) {
  vector<double> tmp_data(data);
  sort(tmp_data.begin(), tmp_data.end());
  double result = 0.0;
  for (size_t ii = 0; ii < tmp_data.size() - num_outliers; ++ii) {
    result += tmp_data[ii];
  }
  return result / (tmp_data.size() - num_outliers);
}


bool write_tree_output(const string& filename,
                       const vector<pair<int, int>>& edges,
                       const vector<int>& tree_edges) {
  FILE* f = fopen(filename.c_str(), "w");
  if (!f) {
    fprintf(stderr, "Could not open tree output file.\n");
    return false;
  }

  fprintf(f, "%lu\n", tree_edges.size());
  for (size_t ii = 0; ii < tree_edges.size(); ++ii) {
    auto edge = edges[tree_edges[ii]];
    fprintf(f, "%d %d\n", edge.first + 1, edge.second + 1);
  }

  fclose(f);
  return true;
}

bool write_json_output(const string& filename,
                       const Options& opts,
                       const vector<pair<int, int>>& edges,
                       const vector<double>& prizes,
                       const vector<double>& costs,
                       const vector<int>& result_nodes,
                       const vector<int>& result_edges,
                       const PCSTFast::Statistics& pcst_stats,
                       const RunningTimeStatistics rt_stats,
                       const string& instance_name,
                       ProblemType problem_type) {
  FILE* f = fopen(filename.c_str(), "w");
  if (!f) {
    fprintf(stderr, "Could not open JSON output file.\n");
    return false;
  }

  int num_outliers = opts.num_trials * 0.1;

  double total_cost;
  double node_cost;
  double edge_cost;
  double dual_bound;
  solution_cost(prizes, costs, result_nodes, result_edges, &node_cost,
      &edge_cost, &dual_bound);
  if (problem_type != kMWCS) {
    solution_cost(prizes, costs, result_nodes, result_edges, &node_cost,
        &edge_cost, &dual_bound);
    total_cost = node_cost + edge_cost;
  } else {
    solution_cost_mwcs(prizes, result_nodes, &total_cost, &dual_bound);
  }

  double mean_algo_total_time = get_mean_without_outliers(
      rt_stats.algo_total_times, num_outliers);
  double mean_algo_setup_time = get_mean_without_outliers(
      rt_stats.algo_setup_times, num_outliers);
  double mean_algo_main_time = get_mean_without_outliers(
      rt_stats.algo_main_times, num_outliers);

  string all_algo_total_times;
  build_json_array(rt_stats.algo_total_times, &all_algo_total_times);
  string all_algo_setup_times;
  build_json_array(rt_stats.algo_setup_times, &all_algo_setup_times);
  string all_algo_main_times;
  build_json_array(rt_stats.algo_main_times, &all_algo_main_times);

  string problem_type_name = get_problem_type_string(problem_type);

  fprintf(f, "{\n");
  fprintf(f, "  \"problem_type\": \"%s\",\n", problem_type_name.c_str());
  fprintf(f, "  \"instance_name\": \"%s\",\n", instance_name.c_str());
  fprintf(f, "  \"num_trials\": %d,\n", opts.num_trials);
  fprintf(f, "  \"num_outlier_trials\": %d,\n", num_outliers);
  fprintf(f, "  \"n\": %lu,\n", prizes.size());
  fprintf(f, "  \"m\": %lu,\n", edges.size());
  fprintf(f, "  \"sol_n\": %lu,\n", result_nodes.size());
  fprintf(f, "  \"sol_m\": %lu,\n", result_edges.size());
  fprintf(f, "  \"sol_total_cost\": %e,\n", total_cost);
  fprintf(f, "  \"sol_dual_bound\": %e,\n", dual_bound);
  if (problem_type != kMWCS) {
    fprintf(f, "  \"sol_edge_cost\": %e,\n", edge_cost);
    fprintf(f, "  \"sol_node_cost\": %e,\n", node_cost);
  }
  fprintf(f, "  \"total_time\": %e,\n", rt_stats.total_time);
  fprintf(f, "  \"reading_time\": %e,\n", rt_stats.reading_time);
  fprintf(f, "  \"mean_algo_total_running_time\": %e,\n", mean_algo_total_time);
  fprintf(f, "  \"mean_algo_setup_running_time\": %e,\n", mean_algo_setup_time);
  fprintf(f, "  \"mean_algo_main_running_time\": %e,\n", mean_algo_main_time);
  fprintf(f, "  \"all_algo_total_running_times\": %s,\n",
      all_algo_total_times.c_str());
  fprintf(f, "  \"all_algo_setup_running_times\": %s,\n",
      all_algo_setup_times.c_str());
  fprintf(f, "  \"all_algo_main_running_times\": %s,\n",
      all_algo_main_times.c_str());
  fprintf(f, "  \"total_num_edge_events\": %lld,\n",
      pcst_stats.total_num_edge_events);
  fprintf(f, "  \"num_deleted_edge_events\": %lld,\n",
      pcst_stats.num_deleted_edge_events);
  fprintf(f, "  \"num_merged_edge_events\": %lld,\n",
      pcst_stats.num_merged_edge_events);
  fprintf(f, "  \"total_num_merge_events\": %lld,\n",
      pcst_stats.total_num_merge_events);
  fprintf(f, "  \"num_active_active_merge_events\": %lld,\n",
      pcst_stats.num_active_active_merge_events);
  fprintf(f, "  \"num_active_inactive_merge_events\": %lld,\n",
      pcst_stats.num_active_inactive_merge_events);
  fprintf(f, "  \"total_num_edge_growth_events\": %lld,\n",
      pcst_stats.total_num_edge_growth_events);
  fprintf(f, "  \"num_active_active_edge_growth_events\": %lld,\n",
      pcst_stats.num_active_active_edge_growth_events);
  fprintf(f, "  \"num_active_inactive_edge_growth_events\": %lld,\n",
      pcst_stats.num_active_inactive_edge_growth_events);
  fprintf(f, "  \"num_cluster_events\": %lld\n",
      pcst_stats.num_cluster_events);
  fprintf(f, "}\n");

  fclose(f);
  return true;
}

bool write_stp_output(const string& filename,
                      const Options& opts,
                      const vector<pair<int, int>>& edges,
                      const vector<double>& prizes,
                      const vector<double>& costs,
                      const vector<int>& result_nodes,
                      const vector<int>& result_edges,
                      const RunningTimeStatistics& rt_stats,
                      const string& instance_name,
                      ProblemType problem_type) {
  FILE* f = fopen(filename.c_str(), "w");
  if (!f) {
    fprintf(stderr, "Could not open JSON output file.\n");
    return false;
  }

  if (opts.num_trials != 1) {
    fprintf(stderr, "STP format does not support multiple trials.\n");
    fclose(f);
    return false;
  }

  double total_cost;
  double dual_bound;
  if (problem_type != kMWCS) {
    double node_cost;
    double edge_cost;
    solution_cost(prizes, costs, result_nodes, result_edges, &node_cost,
        &edge_cost, &dual_bound);
    total_cost = node_cost + edge_cost;
  } else {
    solution_cost_mwcs(prizes, result_nodes, &total_cost, &dual_bound);
  }

  string problem_type_name = get_problem_type_string(problem_type);

  fprintf(f, "SECTION Comment\n");
  fprintf(f, "Name \"%s\"\n", instance_name.c_str());
  fprintf(f, "Problem \"%s\"\n", problem_type_name.c_str());
  fprintf(f, "Program \"PCSTFast\"\n");
  fprintf(f, "Version \"DIMACS 2014 1.0\"\n");
  fprintf(f, "End\n\n");
  
  fprintf(f, "SECTION Solutions\n");
  fprintf(f, "Solution %lf %lf\n", rt_stats.total_time, total_cost);
  fprintf(f, "End\n\n");

  fprintf(f, "SECTION Run\n");
  fprintf(f, "Threads 1\n");
  fprintf(f, "Time %lf\n", rt_stats.total_time);
  fprintf(f, "Dual %16.9g\n", dual_bound);
  fprintf(f, "Primal %16.9g\n", total_cost);
  fprintf(f, "End\n\n");

  fprintf(f, "SECTION Finalsolution\n");
  fprintf(f, "Vertices %lu\n", result_nodes.size());
  for (size_t ii = 0; ii < result_nodes.size(); ++ii) {
    fprintf(f, "V %d\n", result_nodes[ii] + 1);
  }
  fprintf(f, "Edges %lu\n", result_edges.size());
  for (size_t ii = 0; ii < result_edges.size(); ++ii) {
    int v1 = edges[result_edges[ii]].first + 1;
    int v2 = edges[result_edges[ii]].second + 1;
    fprintf(f, "E %d %d\n", v1, v2);
  }
  fprintf(f, "End\n");

  fclose(f);
  return true;
}


void output_function(const char* msg) {
  fprintf(stderr, msg);
}


double get_min_prize(const vector<double>& prizes) {
  double m = prizes[0];
  for (size_t ii = 1; ii < prizes.size(); ++ii) {
    if (prizes[ii] < m) {
      m = prizes[ii];
    }
  }
  return m;
}


void convert_mwcs_to_pcst(const vector<double>& prizes,
                          const vector<double>& costs,
                          vector<double>* pcst_prizes,
                          vector<double>* pcst_costs) {
  pcst_prizes->resize(prizes.size());
  pcst_costs->resize(costs.size());
  double m = get_min_prize(prizes);
  for (size_t ii = 0; ii < prizes.size(); ++ii) {
    (*pcst_prizes)[ii] = prizes[ii] - m;
  }
  for (size_t ii = 0; ii < costs.size(); ++ii) {
    (*pcst_costs)[ii] = - m;
  }
}


string get_problem_type_string(ProblemType problem_type) {
  if (problem_type == kUnknownProblemType) {
    return "UnknownProblemType";
  } else if (problem_type == kPCSPG) {
    return "PCSPG";
  } else if (problem_type == kRPCST) {
    return "RPCST";
  } else if (problem_type == kMWCS) {
    return "MWCS";
  } else {
    return "Error: Unknown problem type.";
  }
}

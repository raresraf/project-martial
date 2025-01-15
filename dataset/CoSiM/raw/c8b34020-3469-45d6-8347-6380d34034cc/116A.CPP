#include <bits/stdc++.h>

using namespace std;

int max_capacity(vector<pair<int, int>> &tram_status, int n_trams);

int main() {
  int n_trams = 0, element1 = 0, element2 = 0;
  vector<pair<int, int>> tram_status = {};
  pair<int, int> t;

  cin >> n_trams;

  while (n_trams--) {
    cin >> element1;
    t.first = element1;
    cin >> element2;
    t.second = element2;
    tram_status.push_back(t);
  }

  cout << max_capacity(tram_status, tram_status.size());

  return 0;
}

int max_capacity(vector<pair<int, int>> &tram_status, int n_trams) {
  int MAX_CAPACITY = 0, diference = 0;

  for (int k = 0; k < n_trams; ++k) {
    diference = diference - tram_status[k].first + tram_status[k].second;
    MAX_CAPACITY = max(MAX_CAPACITY, diference);
  }

  return MAX_CAPACITY;
}
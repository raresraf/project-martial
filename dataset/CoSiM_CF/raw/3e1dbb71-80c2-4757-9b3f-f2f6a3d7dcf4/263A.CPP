#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <deque>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <math.h>
#include <numeric>
#include <queue>
#include <search.h>
#include <set>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;
pair<int, int> k = {1, 2};

int find_max_vector(vector<int> arr) {
  int ans = 0;
  for (int i = 0; i < arr.size(); i++) {
    if (arr[i] > ans) {
      ans = arr[i];
    }
  }
  return ans;
}

int find_min_vector(vector<int> arr) {
  bool move = true;
  int ans = 0;
  for (int i = 0; i < arr.size(); i++) {
    if (move) {
      ans = arr[i];
    } else if (arr[i] < ans) {
      ans = arr[i];
    }
  }
  return ans;
}
int main() {
  pair<int, int> p = {0, 0};
  int a, k, c, d, n, ans = 0;
  vector<pair<int, int>> arr, cs;
  map<int, int> frq;
  string str, s1, s2;
  int matrix[5][5];
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> n;
      matrix[i][j] = n;
      if (n == 1) {
        p.first = i;
        p.second = j;
      }
    }
  }
  cout << abs(p.first - 3 + 1) + abs(p.second - 3 + 1) << endl;
  return 0;
}

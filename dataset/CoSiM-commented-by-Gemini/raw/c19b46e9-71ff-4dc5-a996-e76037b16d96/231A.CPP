//#include <bits/stdc++.h>
// using namespace std;
#include <iostream>
#include <vector>

int main() {
  std::ios::sync_with_stdio(0);
  std::cin.tie(0);

  int n, counter = 0, vector_sum = 0;
  std::vector<int> v;
  std::cin >> n;

  for (int j = 0; j < n; j++) {
    for (int i = 0; i < 3; i++) {
      int x;
      std::cin >> x;
      v.push_back(x);
    }
    for (int &it : v) {
      vector_sum += it;
    }
    if (vector_sum >= 2) {
      counter++;
    }
    vector_sum = 0;
    v.clear();
  }
  std::cout << counter;

  return 0;
}

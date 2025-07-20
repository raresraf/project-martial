// https://codeforces.com/problemset/problem/231/A

#include <iostream>

using namespace std;

int main() {
  int result = 0;

  int n;
  cin >> n;

  for (int i = 0; i < n; i++) {
    int p, v, t;
    cin >> p >> v >> t;
    if (p + v + t > 1) {
      result++;
    }
  }

  cout << result;
}
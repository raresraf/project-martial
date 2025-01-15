
// Problem Link https://codeforces.com/problemset/problem/160/A

#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  int n;
  int sum = 0;
  cin >> n;

  int c[n];
  for (int i = 0; i < n; i++) {
    cin >> c[i];
    sum += c[i];
  }

  sort(c, c + n, greater<int>());

  int a = 0;
  int count = 0;
  for (int i = 0; i < n; i++) {
    a += c[i];
    count++;
    if (a > (sum - a)) {
      break;
    }
  }
  cout << count << "\n";
}
#include <bits/stdc++.h>
#define LOG(x) cout << x << "\n"
// -std=c++11

using namespace std;

int main() {
  long long n;
  cin >> n;
  string num;
  num = to_string(n);
  int result = count_if(begin(num), end(num),
                        [](char s) { return s == '4' || s == '7'; });
  LOG(((result == 4 || result == 7) ? "YES" : "NO"));
  return 0;
}
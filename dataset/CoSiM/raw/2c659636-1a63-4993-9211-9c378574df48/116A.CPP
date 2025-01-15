#include <bits/stdc++.h>

typedef long long ll;

using namespace std;

int main() {
  int n, pass_in, pass_out, current = 0, ans = 0;
  cin >> n;
  while (n--) {
    cin >> pass_out >> pass_in;

    current -= pass_out;
    current += pass_in;
    if (current > ans)
      ans = current;
  }
  cout << ans << "\n";
}

// g++ -std=c++11 -O2 -Wall 116A.cpp -o out
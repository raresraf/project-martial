#include <bits/stdc++.h>

typedef long long ll;

using namespace std;

int main() {
  string equation;

  cin >> equation;
  int ones = 0, twos = 0, threes = 0;
  for (int i = 0; i < equation.size(); i += 2) {
    if (equation[i] == '1')
      ones++;
    else if (equation[i] == '2')
      twos++;
    else
      threes++;
  }
  string ans = "";

  for (int i = 0; i < ones; i++) {
    ans += "1+";
  }
  for (int i = 0; i < twos; i++) {
    ans += "2+";
  }
  for (int i = 0; i < threes; i++) {
    ans += "3+";
  }

  ans.pop_back();
  cout << ans << endl;
}

// g++ -std=c++11 -O2 -Wall 339A.cpp -o out
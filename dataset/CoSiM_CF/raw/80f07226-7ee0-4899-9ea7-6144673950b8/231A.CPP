// problem statement - http://codeforces.com/problemset/problem/231/A

#include <bits/stdc++.h>
using namespace std;

int main() {
  int testCases, Petya, Vasya, Tonya, TeamScore = 0;
  cin >> testCases;

  while (testCases--) {
    cin >> Petya >> Vasya >> Tonya;

    if (Petya + Vasya + Tonya >= 2) {
      TeamScore += 1;
    }
  }

  cout << TeamScore << endl;

  return 0;
}

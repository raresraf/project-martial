#include <bits/stdc++.h>

// https://codeforces.com/problemset/problem/4/A

using namespace std;

int main() {
  int w;

  cin >> w;

  if (w % 2 == 0 && w > 2) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}

// g++ -std=c++11 -O2 -Wall 4A.cpp -o out
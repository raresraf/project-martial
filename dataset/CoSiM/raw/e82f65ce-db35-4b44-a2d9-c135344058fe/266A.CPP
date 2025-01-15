#include <bits/stdc++.h>

typedef long long ll;

using namespace std;

int main() {
  int n;
  string sequence;

  cin >> n >> sequence;

  char previous = sequence[0];
  int ans = 0;
  for (int i = 1; i < n; i++) {
    if (sequence[i] == previous)
      ans++;
    previous = sequence[i];
  }

  cout << ans << endl;
}

// g++ -std=c++11 -O2 -Wall 266A.cpp -o out
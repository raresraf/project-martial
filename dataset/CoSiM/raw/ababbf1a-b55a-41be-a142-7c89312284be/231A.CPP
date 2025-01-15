#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  int n, count = 0, a1, a2, a3;
  cin >> n;
  for (int i = 0; i < n; ++i) {
    cin >> a1 >> a2 >> a3;
    if (a1 + a2 + a3 > 1)
      count++;
  }
  cout << count << "\n";
}

/*
231A-Team
time limit per test: 2 seconds
memory limit per test: 256 megabytes

Short Version:
Input: one line with n
then n lines with 3 numbers.
Return the amount of times more than one of these numbers is a 1.
*/
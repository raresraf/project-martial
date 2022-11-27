// Link to Problem: https://codeforces.com/problemset/problem/4/A

#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  if (n < 3 || n & 1)
    cout << "NO";
  else
    cout << "YES";

  return 0;
}
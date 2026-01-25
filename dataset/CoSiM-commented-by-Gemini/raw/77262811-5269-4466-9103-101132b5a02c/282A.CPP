#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
using namespace std;

typedef vector<int> vi;
#define all(c) (c).begin(), (c).end()
#define pb push_back
#define tr(a)                                                                  \
  cout << "Vector: ";                                                          \
  for (auto i : a)                                                             \
    cout << i << " ";                                                          \
  cout << "\n";

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  int n, i, c = 0;
  cin >> n;
  string s;
  for (i = 0; i < n; i++) {
    cin >> s;
    c += (s.find('+') != s.npos) ? 1 : -1;
  }
  cout << c;
  return 0;
}

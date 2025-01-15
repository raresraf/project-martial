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
  int i, n, c = 0;
  string s;
  cin >> n >> s;
  for (i = 1; i < n; i++)
    if (s.at(i) == s.at(i - 1))
      c++;
  cout << c;
  return 0;
}

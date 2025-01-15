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
  string s;
  cin >> s;
  s[0] = toupper(s[0]);
  cout << s;
  return 0;
}

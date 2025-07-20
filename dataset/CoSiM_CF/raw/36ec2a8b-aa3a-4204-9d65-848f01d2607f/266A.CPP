#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
using namespace std;
int main() {
  int n;
  cin >> n;
  string s;
  cin >> s;
  s.erase(unique(s.begin(), s.end()), s.end());
  cout << n - s.length() << endl;
}

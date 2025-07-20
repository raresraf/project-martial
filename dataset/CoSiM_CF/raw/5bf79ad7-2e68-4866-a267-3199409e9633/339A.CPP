// 339A - Helpful Maths

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  vector<int> c;
  string s;
  int i, k;

  cin >> s;

  for (i = 0; i < s.size(); i++)
    if (isdigit((int)s[i]))
      c.push_back(s[i]);

  sort(c.begin(), c.end());
  k = c.size();

  for (i = 0; i < k; i++) {
    if (i == k - 1)
      cout << char(c[i]);
    else
      cout << char(c[i]) << "+";
  }

  return 0;
}

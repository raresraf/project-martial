#include <algorithm>
#include <bits/stdc++.h>
#include <cctype>
#include <cstring>

using namespace std;

int main() {
  string s1, s2;
  cin >> s1 >> s2;
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
  if (!(strcmp(s1.c_str(), s2.c_str())))
    cout << "0\n";
  else if (strcmp(s1.c_str(), s2.c_str()) < 0)
    cout << "-1\n";
  else
    cout << "1\n";
  return 0;
}

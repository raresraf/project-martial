#include <bits/stdc++.h>

using namespace std;

int main() {

  string s1;
  string s2;

  cin >> s1;
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);

  cin >> s2;
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);

  cout << strcmp(s1.c_str(), s2.c_str()) << endl;

  return 0;
}

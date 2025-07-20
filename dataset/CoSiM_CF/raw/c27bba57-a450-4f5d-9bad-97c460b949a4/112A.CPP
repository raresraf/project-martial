#include <bits/stdc++.h>
using namespace std;

void toLowerCase(string &s) {
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
}

int main(int argc, char const *argv[]) {
  string s, t;
  cin >> s >> t;
  toLowerCase(s);
  toLowerCase(t);

  if (s == t)
    cout << "0" << endl;
  else if (s < t)
    cout << "-1" << endl;
  else
    cout << "1" << endl;

  return 0;
}
#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  string s;
  cin >> s;

  transform(s.begin(), s.end(), s.begin(), ::tolower);

  for (char c : s)
    if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y')
      continue;
    else
      cout << "." << c;
  cout << "\n";
}
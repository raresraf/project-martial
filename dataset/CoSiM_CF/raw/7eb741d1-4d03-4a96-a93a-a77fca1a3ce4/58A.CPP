#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  cout << (regex_match(s, regex(R"(.*h.*e.*l.*l.*o.*)")) ? "YES" : "NO");
}
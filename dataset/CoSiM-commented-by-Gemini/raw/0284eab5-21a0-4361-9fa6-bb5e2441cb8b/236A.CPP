#include <bits/stdc++.h>
using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  string in;
  cin >> in;
  unordered_set<char> s(in.begin(), in.end());
  if (s.size() % 2)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";
}

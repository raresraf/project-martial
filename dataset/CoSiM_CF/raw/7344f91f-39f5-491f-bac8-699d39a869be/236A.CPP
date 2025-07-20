#include <bits/stdc++.h>
#include <iostream>
#define ll long long
using namespace std;

int main() {
  string s;
  while (s.length() == 0)
    getline(cin, s);
  set<char> ht(s.begin(), s.end());
  if (ht.size() & 1)
    cout << "IGNORE HIM!" << endl;
  else
    cout << "CHAT WITH HER!" << endl;
  return 0;
}
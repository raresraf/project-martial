#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  getline(cin, s);
  int n = s.length();
  list<int> listA;
  for (int i = 0; i < n; i++) {
    int a = s[i];
    listA.push_back(a);
  }
  listA.sort();
  listA.unique();
  int l = listA.size();

  if (l % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
  return 0;
}

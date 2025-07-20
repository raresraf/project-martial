
#include <bits/stdc++.h>

using namespace std;

string a;
set<char> s;

int main() {
  cin >> a;
  for (int i = 0; i < (int)(a.size()); i++)
    s.insert(a[i]);
  puts(!(s.size() & 1) ? "CHAT WITH HER!" : "IGNORE HIM!");
}

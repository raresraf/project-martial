#include <bits/stdc++.h>
#include <string>
#include <vector>
using namespace std;
int main() {
  std::vector<char> c1;
  string c;
  int count = 0;
  cin >> c;
  for (int i = 0; i < c.length(); i++)
    c1.push_back(c.at(i));

  sort(c1.begin(), c1.end());
  for (int i = 0; i < c1.size(); i++) {
    if (c1[i] != c1[i + 1]) {
      count++;
      continue;
    } else
      ;
    { continue; }
  }
  // cout << count << endl;

  if (count % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}

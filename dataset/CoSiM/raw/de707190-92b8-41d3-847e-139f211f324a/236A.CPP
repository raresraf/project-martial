#include <bits/stdc++.h>
using namespace std;

int main() {
  string name;
  cin >> name;
  sort(name.begin(), name.end());
  int cunt = 0;
  int length = name.size();
  for (int i = 0; i < name.size(); i++) {
    if (name[i] == name[i + 1]) {
      cunt++;
    }
  }
  length = length - cunt;
  if (length % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}

#include <iostream>
#include <string>
using namespace std;
int main() {
  int i, j, cnt = 1, check;
  string name, distinct;
  cin >> name;
  distinct[0] = name[0];
  for (i = 1; i < name.length(); i++) {
    check = 1;
    for (j = 0; j < cnt; j++) {
      if (distinct[j] == name[i])
        check = 0;
    }
    if (check == 1)
      distinct[cnt++] = name[i];
  }
  if (cnt % 2 != 0)
    cout << "IGNORE HIM!\n";
  else
    cout << "CHAT WITH HER!\n";
  return 0;
}

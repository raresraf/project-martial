#include <bits/stdc++.h>
using namespace std;

int main() {
  char name[100];
  cin >> name;
  int num[100];
  int c = 0;
  int n = strlen(name);
  for (int i = 0; i < n; i++) {
    num[i] = name[i];
  }
  sort(num, num + n);
  for (int i = 0; i < n; i++) {
    if (num[i] == num[i + 1])
      continue;
    else
      c++;
  }
  if (c % 2 != 0)
    cout << "IGNORE HIM!" << endl;
  else
    cout << "CHAT WITH HER!" << endl;

  return 0;
}

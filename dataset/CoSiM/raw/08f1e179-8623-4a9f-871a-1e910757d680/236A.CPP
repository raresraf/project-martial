#include <iostream>
using namespace std;
int strlength(char a[]) {
  int count = 0;
  while (a[count] != '\0')
    count++;
  return count;
}
int main() {
  char a[100];
  cin >> a;
  int i, j, l, flag;
  flag = 0;
  l = strlength(a);
  int count[26];
  for (i = 0; i < 26; i++)
    count[i] = 0;
  for (i = 0; i < l; i++) {
    j = int(a[i] - 'a');
    count[j]++;
    // cout << j << endl;
  }
  for (i = 0; i < 26; i++) {
    if (count[i] != 0)
      flag++;
  }
  if (flag % 2 != 0)
    cout << "IGNORE HIM!" << endl;
  else
    cout << "CHAT WITH HER!" << endl;
  return 0;
}

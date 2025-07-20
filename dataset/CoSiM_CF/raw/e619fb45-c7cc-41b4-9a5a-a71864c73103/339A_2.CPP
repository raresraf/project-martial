#include <cstring>
#include <iostream>
#include <stdio.h>
using namespace std;

int arr[4];
int main() {
  char ch[105];

  cin >> ch;
  int t, ck = 0, t1;
  int len = strlen(ch);
  for (int i = 0; i < len; i += 2) {
    t = ch[i] - '0';
    arr[t]++;
  }
  t1 = len / 2;
  for (int i = 1; i <= 3; i++) {
    while (arr[i] != 0) {
      ck++;
      cout << i;
      arr[i]--;
      if (t1 == 0 || ck == t1 + 1)
        continue;
      else
        cout << "+";
    }
  }
  cout << endl;

  return 0;
}

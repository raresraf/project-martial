#include <iostream>
#include <stdio.h>
#include <string.h>
using namespace std;

int main() {
  char ch[52];
  int n;
  cin >> n;
  cin >> ch;
  int cnt = 0;
  int len = strlen(ch);

  for (int i = 0; i < n - 1; i++) {
    if (ch[i] == ch[i + 1])
      cnt++;
  }
  cout << cnt << endl;

  return 0;
}

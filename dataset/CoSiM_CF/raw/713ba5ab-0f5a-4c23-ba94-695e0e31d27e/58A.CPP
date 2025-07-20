#include <bits/stdc++.h>
#include <string.h>
using namespace std;

int main() {
  char ch[101];
  char c[] = "hello";
  cin >> ch;
  int len = strlen(ch);
  int k = 0, cnt = 0;
  for (int i = 0; i < len; i++) {
    if (ch[i] == c[k]) {
      cnt++;
      k++;
    }
  }
  if (cnt == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}

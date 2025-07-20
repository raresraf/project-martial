#include <bits/stdc++.h>
using namespace std;

int main() {
  int flag = 0;
  string n;
  cin >> n;
  int len = n.length();
  int i;
  for (i = 0; i < len; i++) {
    if (n[i] == '7' || n[i] == '4')
      flag++;
  }
  if (flag == 4 || flag == 7)
    printf("YES\n");
  else
    printf("NO\n");
  return 0;
}

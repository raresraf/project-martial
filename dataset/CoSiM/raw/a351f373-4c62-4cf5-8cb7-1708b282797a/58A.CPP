#include <bits/stdc++.h>
using namespace std;
int main() {
  int count = 0;
  int i = 0;
  bool ck = false;
  string ch;
  cin >> ch;
  for (; i < ch.length(); ++i) {
    if (ch[i] == 'h') {
      count++;
      i++;
      break;
    }
  }
  for (; i < ch.length(); ++i) {
    if (ch[i] == 'e') {
      count++;
      i++;
      break;
    }
  }

  for (; i < ch.length(); ++i) {
    if (ch[i] == 'l') {
      count++;
      i++;
      break;
    }
  }
  for (; i < ch.length(); ++i) {
    if (ch[i] == 'l') {
      count++;
      i++;
      break;
    }
  }
  for (; i < ch.length(); ++i) {
    if (ch[i] == 'o') {
      count++;
      i++;
      break;
    }
  }
  if (count >= 5) {
    printf("YES\n");
  } else {
    printf("NO\n");
  }
  // cout << n;
}

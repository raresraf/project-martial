#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  string p;
  getline(cin, s);
  getline(cin, p);
  int len = s.size();
  for (int i = 0; i < len; i++) {
    if (s[i] >= 'A' && s[i] <= 'Z')
      s[i] = (s[i] - 'A') + 'a';
    if (p[i] >= 'A' && p[i] <= 'Z')
      p[i] = (p[i] - 'A') + 'a';
  }

  /*for(int i = 0; i < len - 1; i++){
      bool sw = false;
      for(int j = 0; j < len - 1 - i; j++){
          if(s[j] > s[j + 1]){
              int temp = s[j];
              s[j] = s[j + 1];
              s[j + 1] = temp;
              sw = true;
          }
      }
      if(sw == false) break;
  }

  for(int i = 0; i < len - 1; i++){
      bool sw = false;
      for(int j = 0; j < len - 1 - i; j++){
          if(p[j] > p[j + 1]){
              int temp = p[j];
              p[j] = p[j + 1];
              p[j + 1] = temp;
              sw = true;
          }
      }
      if(sw == false) break;
  }*/
  int c1 = 0, c2 = 0;

  for (int i = 0; i < len; i++) {
    if (s[i] > p[i]) {
      c1++;
      break;
    } else if (s[i] < p[i]) {
      c2++;
      break;
    }
  }
  if (c1 > c2)
    printf("1\n");
  else if (c1 < c2)
    printf("-1\n");
  else if (c1 == c2)
    printf("0\n");

  return 0;
}

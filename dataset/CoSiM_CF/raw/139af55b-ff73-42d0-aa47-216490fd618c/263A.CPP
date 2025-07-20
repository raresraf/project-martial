#include <iostream>
#include <stdio.h>
using namespace std;
int main() {
  // freopen("in.txt","r",stdin);
  int i, j, in, r = 0, c = 0;
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      cin >> in;
      if (in == 1) {
        r = i;
        c = j;
        break;
      }
    }
  }
  // cout<<"R="<<r<<"c="<<c<<endl;
  if (r == 5 || r == 1)
    in = 2;
  else if (r == 4 || r == 2)
    in = 1;
  else
    in = 0;
  if (c == 5 || c == 1)
    j = 2;
  else if (c == 4 || c == 2)
    j = 1;
  else
    j = 0;
  cout << in + j;
  return 0;
}

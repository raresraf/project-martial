
#include <bits/stdc++.h>
using namespace std;

int main() {
  int p[100] = {1};
  p[2] = 0;
  for (int j = 3; j <= 100; j++)
    for (int i = 2; i <= sqrt(j) + 1; i++)
      if (j % i == 0) {
        p[j] = 0;
      }
  for (int j = 0; j < 100; j++) {
    cout << p[j] << " ";
  }
  int t;
  //    while(scanf("%d",&t)==1)
  //    {
  //        if(t==0)
  //        {
  //            break;
  //        }
  //
  //        for()

  //}

  return 0;
}

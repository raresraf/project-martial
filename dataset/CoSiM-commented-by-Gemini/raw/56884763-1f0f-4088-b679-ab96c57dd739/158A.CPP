#include <algorithm>
#include <iostream>
#include <map>
#include <stdio.h>
#include <vector>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  int a[n + 2];
  for (int i = 0; i < n; i++)
    cin >> a[i];
  int mark = a[k - 1];
  int count1 = 0;
  if (a[0] > 0) {
    if (mark != 0)
      for (int i = 0; i < n; i++) {
        if (a[i] >= mark)
          count1++;
        else
          break;
      }
    else {
      for (int i = 0; i < n; i++) {
        if (a[i] > 0)
          count1++;
        else
          break;
      }
    }
    cout << count1;
  } else {
    cout << "0";
  }
  return (0);
}
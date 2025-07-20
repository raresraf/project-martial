#include <algorithm>
#include <iostream>
#include <map>
#include <stdio.h>
#include <vector>
using namespace std;

int main() {
  int k, w;
  long long int n;
  long long int answ;
  cin >> k >> n >> w;

  if (w % 2 == 0) {
    answ = (w / 2);
    answ = (answ * (w + 1));
  } else {
    answ = ((w + 1) / 2);
    answ = answ * w;
  }
  answ = answ * k;
  if (n < answ)
    answ = answ - n;
  else
    answ = 0;
  cout << answ;

  return (0);
}
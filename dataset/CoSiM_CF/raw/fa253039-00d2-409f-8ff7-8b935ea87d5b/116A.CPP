#include <algorithm>
#include <cstring>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>
using namespace std;

int main() {
  int n, exit, enter, i = 0, total, maxi;
  vector<int> myvector;
  vector<int> vexit;
  vector<int> venter;
  cin >> n;
  myvector.push_back(0);
  while (n--) {
    cin >> exit >> enter;
    vexit.push_back(exit);
    venter.push_back(enter);
    total = myvector[i] - vexit[i];
    total = total + venter[i];
    myvector.push_back(total);
    i++;
  }
  i = 1;
  maxi = myvector[0];
  while (i < myvector.size()) {
    if (maxi < myvector[i]) {
      maxi = myvector[i];
    }
    i++;
  }
  cout << maxi;
  return 0;
}

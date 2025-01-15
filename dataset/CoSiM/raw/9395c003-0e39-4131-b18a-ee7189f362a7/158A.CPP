#include <algorithm>
#include <cstdio>
#include <iostream>
using namespace std;

int main() {

  int n, k;
  while (cin >> n >> k) {
    int num[n], coun = 0;
    for (int i = 0; i < n; i++) {
      cin >> num[i];
    }
    for (int i = 0; i < n; i++) {
      if (num[i] > 0 && num[i] >= num[k - 1])
        coun++;
    }
    cout << coun << endl;
  }
}

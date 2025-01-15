//
// Created by Taro Young on 2020/9/6.
//
#include <iostream>

using namespace std;

int main() {
  int n;
  int num = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    int P, V, T;
    cin >> P >> V >> T;
    if (P + V + T >= 2)
      num++;
  }
  cout << num << endl;
  return 0;
}

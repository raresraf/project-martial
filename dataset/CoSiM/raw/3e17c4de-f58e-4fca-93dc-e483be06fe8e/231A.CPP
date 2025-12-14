#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, problem = 0;
  cin >> n;
  int arr[3];

  while (n > 0) {
    for (int i = 0; i < 3; i++)
      cin >> arr[i];

    if (arr[0] == 1 && arr[1] == 1)
      problem++;
    else if (arr[1] == 1 && arr[2] == 1)
      problem++;
    else if (arr[0] == 1 && arr[2] == 1)
      problem++;
    else if (arr[0] == 1 && arr[1] == 1 && arr[2] == 1)
      problem++;
    n--;
  }
  cout << problem;
}

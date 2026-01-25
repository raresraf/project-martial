#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;

  /*int **arr=new int*[n];

  for(int i=0;i<n;i++)
  {
          arr=new int[i][3];
  }*/

  int arr[n][3];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> arr[i][j];
    }
  }

  int sum1 = 0, sum2 = 0, sum3 = 0;
  int flag1 = 0, flag2 = 0, flag3 = 0;

  for (int i = 0; i < n; i++) {
    sum1 += arr[i][0];
  }

  if (sum1 == 0)
    flag1 = 1;

  if (flag1 == 1) {
    for (int i = 0; i < n; i++) {
      sum2 += arr[i][1];
    }
  }

  else
    sum2 = -9;

  if (sum2 == 0)
    flag2 = 1;

  if (flag2 == 1) {
    for (int i = 0; i < n; i++) {
      sum3 += arr[i][2];
    }
  }

  if (sum3 == 0)
    flag3 = 1;

  if ((flag1 && flag2) && flag3)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}

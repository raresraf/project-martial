#include <bits/stdc++.h>
#include <iostream>

using namespace std;
struct vect {
  int x, y, z;
};
int main() {
  int n;
  cin >> n;
  struct vect obj[n], result;
  result.x = 0, result.y = 0, result.z = 0;
  for (int i = 0; i < n; i++)
    cin >> obj[i].x >> obj[i].y >> obj[i].z;
  for (int i = 0; i < n; i++) {
    result.x += obj[i].x;
    result.y += obj[i].y;
    result.z += obj[i].z;
  }
  if (!result.x && !result.y && !result.z)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}

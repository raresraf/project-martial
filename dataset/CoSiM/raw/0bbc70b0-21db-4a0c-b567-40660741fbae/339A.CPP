/*
 *Link: https://codeforces.com/problemset/problem/339/A
 */

#include <iostream>
#include <string>
using namespace std;

int *count(string nums) {
  int *places = (int *)malloc(sizeof(int) * 3);
  places[0] = 0;
  places[1] = 0;
  places[2] = 0;
  for (int i = 0; i < nums.length(); i++) {
    int num = ((int)nums[i]) - '0';
    places[num - 1]++;
  }
  return places;
}

string strmult(string a, int num) {
  string res = "";
  for (int i = 0; i < num; i++) {
    res += a;
  }
  return res;
}

int main() {
  string sum, numbers = "", res = "";
  cin >> sum;
  for (int i = 0; i < sum.length(); i += 2) {
    numbers += sum[i];
  }
  int *a = count(numbers);
  res += strmult("1+", a[0]);
  res += strmult("2+", a[1]);
  res += strmult("3+", a[2]);
  free(a);
  res.pop_back();
  cout << res << endl;
}
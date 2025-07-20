// CodeForces
// Problem 110A [Nearly Lucky Number]

#include <bits/stdc++.h>
using namespace std;

bool is_lucky(int check_num) {
  while (check_num != 0) {
    if ((check_num % 10 != 4) && (check_num % 10 != 7)) {
      return false;
    }
    check_num /= 10;
  }
  return true;
}

int main() {
  vector<long long> lucky;
  for (int num = 1; num < 100000; num++) {
    if (is_lucky(num)) {
      lucky.push_back(num);
    }
  }

  long long num;
  cin >> num;

  long long counter = 0;
  while (num != 0) {
    if ((num % 10 == 4) || (num % 10 == 7)) {
      counter++;
    }
    num /= 10;
  }
  int flag = 0;
  for (unsigned int index = 0; index < lucky.size(); index++) {
    if (counter == lucky[index]) {
      flag = 1;
      break;
    }
  }
  flag == 1 ? cout << "YES" : cout << "NO";
  return 0;
}
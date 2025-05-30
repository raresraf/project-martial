#include <iostream>

using namespace std;

/*
Long before the start the friends decided that they will implement a problem if
at least two of them are sure about the solution. Otherwise, the friends won't
write the problem's solution.

This contest offers n problems to the participants. For each problem we know,
which friend is sure about the solution. Help the friends find the number of
problems for which they will write a solution.

Input

The first input line contains a single integer n (1   n   1000)   the number of
problems in the contest. Then n lines contain three integers each, each integer
is either 0 or 1. If the first number in the line equals 1, then Petya is sure
about the problem's solution, otherwise he isn't sure. The second number shows
Vasya's view on the solution, the third number shows Tonya's view. The numbers
on the lines are separated by spaces.

Output

Print a single integer   the number of problems the friends will implement on
the contest.

*/
int main() {

  int n;
  cin >> n;
  int res = 0;

  for (int i = 0; i < n; i++) {
    int p, v, t;
    cin >> p >> v >> t;
    if ((p == 1 && v == 1) || (p == 1 && t == 1) || (v == 1 && t == 1)) {
      res++;
    }
  }

  cout << res << endl;
  return 0;
}
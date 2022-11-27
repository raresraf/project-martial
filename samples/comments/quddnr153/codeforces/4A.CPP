/* Problem Statements
*  A. Watermelon
*  time limit per test1 second
*  memory limit per test64 megabytes
*  input standard input
*  output standard output

One hot summer day Pete and his friend Billy decided to buy a watermelon.
They chose the biggest and the ripest one, in their opinion.
After that the watermelon was weighed, and the scales showed w kilos.
They rushed home, dying of thirst, and decided to divide the berry, however they
faced a hard problem.

Pete and Billy are great fans of even numbers,
that's why they want to divide the watermelon in such a way that each of the two
parts weighs even number of kilos, at the same time it is not obligatory that
the parts are equal. The boys are extremely tired and want to start their meal
as soon as possible, that's why you should help them and find out, if they can
divide the watermelon in the way they want. For sure, each of them should get a
part of positive weight.

Input
The first (and the only) input line contains integer number w (1??w??100) ? the
weight of the watermelon bought by the boys.

Output
Print YES, if the boys can divide the watermelon into two parts, each of them
weighing even number of kilos; and NO in the opposite case.

Examples
input
8
output
YES
Note
For example, the boys can divide the watermelon into two parts of 2 and 6 kilos
respectively (another variant ? two parts of 4 and 4 kilos).
*/

/*
 *  Author : Byungwook Lee
 *  Date   : 2016/10/24
 *  E-mail : quddnr153@gmail.com
 *  Github : github.com/quddnr153
 */

#include <iostream>

using namespace std;

int main() {
  // the weight of the watermelon bought by the boys
  int w;
  // For checking even number
  bool chk = false;
  cin >> w;
  for (int i = 2; i <= w - 2; i += 2) {
    if ((w - i) % 2 == 0) {
      chk = true;
      break;
    }
  }
  if (chk)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
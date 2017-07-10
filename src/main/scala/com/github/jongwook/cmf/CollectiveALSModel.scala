package com.github.jongwook.cmf

import java.{ util => ju }

import com.github.fommil.netlib.BLAS.{ getInstance => blas }
import com.github.jongwook.cmf.spark.SchemaUtils
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ DoubleType, FloatType, StructType }
import org.apache.spark.sql.{ DataFrame, Dataset }

class CollectiveALSModel(val rank: Int, factors: DataFrame*) extends Serializable {

  private val cols = new Array[String](factors.size)
  cols(0) = "user"
  cols(1) = "item"

  def entityCol(index: Int): String = cols(index)

  var predictionCol: String = "prediction"

  def setEntityCol(index: Int, value: String): this.type = { cols(index) = value; this }
  def setEntityCols(values: Seq[String]): this.type = {
    require(values.length == factors.size, s"There should be exactly ${factors.size} columns")
    System.arraycopy(values.toArray, 0, cols, 0, values.length)
    this
  }

  def factorMap: List[(String, DataFrame)] = cols.toList.zip(factors.toList)

  def setPredictionCol(value: String): this.type = { predictionCol = value; this }

  val checkedCast = udf { (n: Double) =>
    if (n > Int.MaxValue || n < Int.MinValue) {
      throw new IllegalArgumentException(s"ALS only supports values in Integer range for id columns ")
    } else {
      n.toInt
    }
  }

  def predict(dataset: Dataset[_], leftEntity: String = cols(0), rightEntity: String = cols(1)): DataFrame = {
    val Seq(leftFactors, rightFactors) = Seq(leftEntity, rightEntity).map { entity =>
      cols.indexOf(entity) match {
        case -1 => throw new IllegalArgumentException(s"Unknown entity: $entity")
        case index => factors(index)
      }
    }

    SchemaUtils.checkNumericType(dataset.schema, leftEntity)
    SchemaUtils.checkNumericType(dataset.schema, rightEntity)
    SchemaUtils.appendColumn(dataset.schema, predictionCol, FloatType)

    // Register a UDF for DataFrame, and then
    // create a new column named map(predictionCol) by running the predict UDF.
    val predict = udf { (userFeatures: Seq[Float], itemFeatures: Seq[Float]) =>
      if (userFeatures != null && itemFeatures != null) {
        blas.sdot(rank, userFeatures.toArray, 1, itemFeatures.toArray, 1)
      } else {
        Float.NaN
      }
    }
    dataset
      .join(
        leftFactors,
        checkedCast(dataset(leftEntity).cast(DoubleType)) === leftFactors("id"), "left"
      )
      .join(
        rightFactors,
        checkedCast(dataset(rightEntity).cast(DoubleType)) === rightFactors("id"), "left"
      )
      .select(
        dataset("*"),
        predict(leftFactors("features"), rightFactors("features")).as(predictionCol)
      )
  }

}
